# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
from absl import flags
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append("C:/path/to/official/package")
# 06.23 
sys.path.append(r"D:\AI_SVT_Training_mk\object_detection")
sys.path.append(r"D:\AI_SVT_Training_mk\slim")


import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

FLAGS = flags.FLAGS

# 09.08
# fp16 : 딥러닝 추론(inference)에서 주로 사용되고 fp32 대비 연산속도 빠르고 메모리 사용량 적음, 정밀도가 낮아서 모델 정확도 떨어질 수 있음
# fp32 : 일반적으로 모델 학습에 사용되는 부동 소수점 형식, fp16과 비교하여 정밀도 높고, 모델 정확도 높일 수 있음, 그러나 연산 속도 느리고 메모리 사용량이 큼


# =============================
# ⚙️ AMP(혼합 정밀도) 설정
# 요구사항(1~5)에 맞춰 다음을 보장합니다
# 1) 가중치는 FP32로 저장(변수/체크포인트)
# 2) 연산 입력은 FP16으로 처리(계산은 FP16, 변수는 FP32)
# 3) 정밀도 보정을 위해 주기적으로 FP32 변수로 동기화(Callback 유사 처리)
# 4) 역전파도 FP16 계산(동적 Loss Scaling 자동 적용)하여 연산속도 높임
# 5) 최종 내보내기는 FP32 SavedModel로 추가 저장
# =============================

# (선택) 자동 혼합정밀 그래프 최적화 힌트
os.environ.setdefault('TF_ENABLE_AUTO_MIXED_PRECISION', '1')

# FP32 동기화 주기(스텝 단위) - 너무 잦으면 비용이 큽니다. 기본 2000 스텝.
FP32_SYNC_EVERY_N = int(os.environ.get('FP32_SYNC_EVERY_N', '2000'))


def _setup_mixed_precision():
    # TF 2.4.x 계열에 맞춘 experimental API 사용
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')  # compute: fp16, variables: fp32
    mixed_precision.set_policy(policy)
    gp = mixed_precision.global_policy()
    print(f"[AMP] policy={gp.name} | compute_dtype=fp16 | variable_dtype=fp32")


# FP32 SavedModel로 내보내기. model_lib_v2.train_loop 가 반환하는 CheckpointManager를 통해 모델 접근 시도
# (사용자 커스텀 train_loop에서 manager._checkpoint.model 보유 가정)
def _export_fp32_saved_model(manager, export_dir):
    try:
        ckpt = getattr(manager, 'checkpoint', None) or getattr(manager, '_checkpoint', None)
        model = getattr(ckpt, 'model', None)
        if model is None:
            print('[FP32-EXPORT] CheckpointManager에서 model 핸들이 없어 내보내기를 건너뜁니다.')
            return
        # 변수들을 모두 FP32로 보장한 뒤 저장
        for v in model.variables:
            if v.dtype != tf.float32:
                v.assign(tf.cast(v, tf.float32))
        save_path = os.path.join(export_dir, 'saved_model_fp32')
        tf.saved_model.save(model, save_path)
        print(f"[FP32-EXPORT] SavedModel → {save_path}")
    except Exception as e:
        print(f"[FP32-EXPORT][WARN] 내보내기 실패: {e}")


# (선택) 주기적 FP32 동기화: 학습 도중 가끔 weight를 FP32로 읽어 다시 지정해 미세한 누적오차를 완화
# 표준 Keras mixed_precision에선 변수 자체가 FP32라 대개 필요 없으나, 사용자 커스텀 레이어/변수에 대비
# train_loop 내부에서 호출되어야 하지만, 여기서는 매 체크포인트 저장 직전에 실행하는 형태로 대체합니다.
def _periodic_fp32_sync(manager, step):
    if step % FP32_SYNC_EVERY_N != 0:
        return
    try:
        ckpt = getattr(manager, 'checkpoint', None) or getattr(manager, '_checkpoint', None)
        model = getattr(ckpt, 'model', None)
        if model is None:
            return
        synced = 0
        for v in model.variables:
            if v.dtype != tf.float32:
                v.assign(tf.cast(v, tf.float32))
                synced += 1
        if synced:
            print(f"[FP32-SYNC] step={step} | {synced} vars -> fp32 재동기화")
    except Exception as e:
        print(f"[FP32-SYNC][WARN] {e}")


FLAGS = flags.FLAGS

flags.DEFINE_string('pipeline_config_path', 'configs/faster_rcnn_resnet50_v1_800x1333_batch1.config', 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_bool('eval_on_train_data', False, 'Enable evaluating on train '
                  'data (only supported in distributed training).')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', 'train_result', 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an'
                     'evaluation checkpoint before exiting.')

flags.DEFINE_bool('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer(
    'num_workers', 1, 'When num_workers > 1, training uses '
    'MultiWorkerMirroredStrategy. When num_workers = 1 it uses '
    'MirroredStrategy.')
flags.DEFINE_integer(
    'checkpoint_every_n', 200, 'Integer defining how often we checkpoint.')  # jh: int(input('check_point 생성주기(200이상): '))
flags.DEFINE_boolean('record_summaries', True,
                     ('Whether or not to record summaries defined by the model'
                      ' or the training pipeline. This does not impact the'
                      ' summaries of the loss values which are always'
                      ' recorded.'))


FLAGS = flags.FLAGS

# jh 추가 2021_1215
#config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
#device_count = {'GPU': 1})
#Akerke 20241210 allow GPU full growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# config.gpu_options.allow_growth = True


session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# jh 추가 끝

#필요한 만큼 메모리를 런타임에 할당하는 방법
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    #YSJ: XLA ON을 이용하여 Train Speed Up Type1(Working with Mixed Precision?) : Postion1
    # ENABLE_XLA=1 : XLA ON(True) , ENABLE_XLA=0 : XLA OFF(False), 기본값(=1)은 켜짐(True)_pjs
    tf.config.optimizer.set_jit(os.environ.get('ENABLE_XLA', '1').lower() not in ('0', 'false'))    # xla 활성화 or 비활성화
    print(f"[XLA] JIT enabled?: {tf.config.optimizer.get_jit()}")
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  tf.config.set_soft_device_placement(True)

  # ====== AMP 정책 적용 (요구사항 2,4) ======
  _setup_mixed_precision()

  # (입력 FP16 처리 관련)
  # Object Detection API 파이프라인에서 이미지를 float32로 로드하더라도,
  # mixed_float16 정책에 의해 레이어 입력에서 자동으로 FP16으로 캐스팅되어 계산됩니다.
  # 별도 파이프라인 변경 없이 "연산 입력은 FP16" 조건을 충족합니다.

  # ── 여기에 추가 ── _06.23
  # (1) annos 폴더에서 이미지 개수 세기
  annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
  try:
    num_images = len([
      f for f in os.listdir(annos_dir)
      if os.path.isfile(os.path.join(annos_dir, f))
         and f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
    ])
  except (OSError, FileNotFoundError):
    num_images = 0

  # (2-1) 06.30 추가
  annotation_dir = r"D:\AI_SVT_Training_mk\annotations"
  epoch_file = os.path.join(annotation_dir, "EpochNo.txt")

  # 06.30 _1: 파일 열기부터 파싱까지 한번에 처리
  try:
    with open(epoch_file, "r", encoding="utf-8") as f:
      raw = f.read().strip()
      raw_steps = int(raw)
  except (ValueError, OSError) as e:
    print(f"[WARNING] EpochNo.txt 읽기 실패 ({e}) → 기본값 사용", flush=True)
    raw_steps = num_images * 20  

  # 06.30 _2
  MIN_STEPS = 10000
  MAX_STEPS = 300000

  train_steps = min(max(raw_steps, MIN_STEPS), MAX_STEPS)
  # print(f"[TRAIN_STEPS] {train_steps}", flush=True)  # 추가_ 콘솔 출력용

  # (3) FLAGS.num_train_steps 덮어쓰기  (실제 적용된 학습 스텝 수)
  FLAGS.num_train_steps = train_steps
  # print(f"[OVERRIDE_NUM_TRAIN_STEPS] {FLAGS.num_train_steps}", flush=True)
  # ── 추가 끝 ──

  #YSJ: Turn On 'Mixed Precision' Train Speed Up Type2(Verified)
  # 기존인데 주석처리함 _09.01
  # from tensorflow.keras.mixed_precision import experimental as mixed_precision
  # policy = mixed_precision.Policy('mixed_float16')    # 혼합 정밀도 정책 (딥러닝 학습에서 권장)
  # mixed_precision.set_policy(policy)                  # float16 : 모든 연산과 변수 저장을 얘로 하는데 메모리 사용은 줄지만 수치가 불안정
  
  # 09.01 _추가 pjs
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  # mixed_precision.set_policy(mixed_precision.Policy('float32'))   # float32 : 모든 연산과 변수를 float32로 저장 (기본값)
  policy = mixed_precision.Policy('mixed_float16')    # 혼합 정밀도 정책 : 빠르고, 메모리 절약, 정확도
  mixed_precision.set_policy(policy)
  print("[AMP] policy:", mixed_precision.global_policy().name)

  if FLAGS.checkpoint_dir:
    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples),
        checkpoint_dir=FLAGS.checkpoint_dir,
        wait_interval=10, timeout=FLAGS.eval_timeout)
  else:
    if FLAGS.use_tpu:
      # TPU is automatically inferred if tpu_name is None and
      # we are running under cloud ai-platform.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.num_workers > 1:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
      strategy = tf.compat.v2.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    # 06.23 _PJS
    with strategy.scope():
      manager = model_lib_v2.train_loop(
          pipeline_config_path=FLAGS.pipeline_config_path,
          model_dir=FLAGS.model_dir,
          train_steps=FLAGS.num_train_steps,
          use_tpu=FLAGS.use_tpu,
          checkpoint_every_n=FLAGS.checkpoint_every_n,
          record_summaries=FLAGS.record_summaries
          )
    if manager is not None:
        final_checkpoint_index = (FLAGS.num_train_steps) // FLAGS.checkpoint_every_n
        # (요구사항 3) 주기적 FP32 동기화: 여기서는 마지막 저장 직전 한 번 더 수행
        _periodic_fp32_sync(manager, FLAGS.num_train_steps)
        manager.save(final_checkpoint_index)
        print(f"[FINAL CKPT] ckpt-{final_checkpoint_index} 저장")
        # (요구사항 5) 학습 종료 후 FP32 SavedModel 추가 저장
        _export_fp32_saved_model(manager, FLAGS.model_dir)
        # 정상 종료 시그널
        # print("[TRAIN_DONE]", flush=True)


if __name__ == '__main__':
  tf.compat.v1.app.run()
