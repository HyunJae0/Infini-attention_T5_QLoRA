# US Stock Market News Sentiment Classification
이 프로젝트의 목표는 미국 증시 뉴스 내용을 기반으로, 해당 뉴스에서 언급된 기업 또는 특정 시장 상황을 약세(Bearish), 중립(Neutral), 강세(Bullish)로 분류하는 미국 증시 도메인 특화 모델을 구축하는 것입니다.

사용 가능한 GPU 메모리가 한정되어 있기 때문에, 공개된 대규모 사전학습 모델 대신 작은 크기의 사전학습 모델을 구축하고자 하였으며, 최종적으로 T5_samll 크기의 사전학습 모델을 만들었습니다.

이 사전학습 모델을 이용하여 두 가지 방식의 파인튜닝을 진행했습니다. 
- 첫 번째는 일반적인 classification task에서 많이 사용되는 방법으로, 모델에 classification head layer를 추가하여 파인튜닝하는 방식입니다.
- 두 번째는 T5처럼 text-to-text로 파인튜닝하는 방식입니다. 이를 위해 'sentiment: '라는 prefix를 사용했습니다.

## 1. datasets
파인튜닝에 사용할 텍스트 데이터셋은 야후 파이낸셜 뉴스이며, 각 뉴스에 Bearish/Neutral/Bullish 레이블이 붙어 있는 지도학습 데이터셋입니다. (https://huggingface.co/datasets/ugursa/Yahoo-Finance-News-Sentences/viewer/default/train?views%5B%5D=train&row=10)

사전학습에서도 파인튜닝 데이터셋과 동일한 도메인 데이셋(블룸버그 파이낸셜 뉴스)을 사용하여, 주식/금융 분야에서 자주 등장하는 복합어 및 특수 용어를 모델에 노출시킴으로써, 해당 도메인 언어 패턴에 익숙하게 만들어 이후 파인튜닝 단계에서 성능 향상을 기대할 수 있습니다. (https://huggingface.co/datasets/genloop/bloomberg_financial_news_120k)

두 데이터셋에 모두 휴리스틱한 텍스트 전처리를 진행하였습니다. 

비지도 사전학습을 위한 데이터셋 처리에 대한 내용은 <code>prepare_unsupervised_dataset_for_pretraining.ipynb</code>에서, 파인튜닝을 위한 데이터셋 처리에 대한 내용은 <code>prepare_supervised_dataset_for_finetuning.ipynb</code>에서 확인할 수 있습니다. 

## 2. pre-training
### 2.1 T5
사전학습 모델은 Google에서 발표한 "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" 논문(https://arxiv.org/abs/1910.10683)에서 제안된 T5(Text-to-Text Transfer Transformer) 모델의 방법론을 기반으로 합니다. 

1. 시퀀스 패킹(sequence packing)

- 논문에서는 패딩 토큰으로 인한 연산 낭비를 최소화하기 위해 시퀀스 패킹을 적용하였습니다.
- 이 프로젝트에서도 이러한 낭비를 줄이고자, 여러 입력 토큰 ID를 이어 붙여 시퀀스 최대 길이(512)를 채우는 방식을 사용했습니다. <code>prepare_unsupervised_dataset_for_pretraining.ipynb</code>

2. 상대 위치 인코딩(Relative Position Encoding)
- 절대 위치 인코딩은 추론 시, 학습 과정에서 보지 못했던 시퀀스 길이가 입력으로 들어오면 해당 위치 정보를 잘 처리하지 못합니다. 즉, 학습한 범위를 벗어나는 길이에 대해서는 일반화 성능이 떨어진다는 단점이 있습니다. 
- T5에서는 이러한 절대 위치 인코딩 대신, 간단한 상대 위치 인코딩 방식을 사용합니다. 
- 어텐션 스코어 행렬에 쿼리 토큰과 키 토큰 간의 상대적 거리에 다른 편향을 직접 더해주는 방식으로 간략하게 설명하면,
- 먼저, 쿼리와 키 시퀀스의 길이(토큰 수)를 바탕으로 쿼리, 키 쌍에 대한 상대 거리 행렬을 생성합니다.
- 이 상대 거리 값들을 그대로 사용하면 다양한 거리 값들이 사용되기 때문에, 가까운 거리는 고유한 값으로(예: 1~7)먼 거리(예: 8 이상)에 대해서는 로그 스케일을 적용합니다.
- 그리고 이 값들을 제한된 제한된 개수의 그룹(버킷, bucket)으로 묶습니다.
- 최종적으로, 임베딩 테이블에서 버킷 ID에 해당하는 편향 값을 가져와서 어텐션 스코어에 더해줍니다. 
- <code>T5_Slim_Attention/slim_attention_and_relative_position_bias.py</code>의 <code>Attention</code> 클래스의 <code>_get_relative_position_bucket</code> 함수와  <code>_compute_bias</code> 함수

3. 사전학습 목적 함수
- 사전학습에는 T5 논문에서 제안한 replace corrupted spans objective를 사용했습니다.
-  이 방식은 아래 그림처럼 입력 텍스트에서 연속된 토큰들의 span을 무작위로 선택한 다음, 선택된 각 스팬을 sentinel 토큰 하나로 대체하여 모델의 입력으로 사용합니다. 모델은 이 입력을 받아, loss를 계산하여 학습을 진행합니다.

<div align="center">
  <img width="350" height="150" alt="image" src="https://github.com/user-attachments/assets/40b5a0d0-a921-404c-b5d5-4befa2788d5f" />
</div>

- <code>T5_Slim_Attention/T5_Slim_Attention/span_corruption.py</code>와 <code>run_pretraining.py</code>

4. Multi-task Learning 
- T5는 unsupervised task와 supervised task를 같이 사전학습하는 방식으로 multi-task pre-training을 사용했습니다.
- 그러나 이 프로젝트에서 사용하는 supervised task는 single task이기 때문에, multi-task pre-training은 적합하지 않다고 판단하여 unsupervised task에 대해서만 사전학습을 진행하였습니다. 

5. 기타
- Dynamic Masking: 모델이 매 에폭마다 새로운 형태의 입력 데이터를 학습할 수 있도록 dynamic masking을 사용하였습니다.
- <code>T5_Slim_Attention/run_pretraining.py</code>의 <code>create_t5_pretraining_data</code> 함수

- AMP(Automatic Mixed Precision): 학습 시 메모리 사용량을 줄이기 위해 PyTorch의 AMP를 사용하였습니다. <code>T5_Slim_Attention/run_pretraining.py</code>의 <code>train</code> 함수
- 참고: https://yjoonjang.medium.com/mixed-precision-training%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-mp-amp-torch-cuda-amp-15c99488ed34

- Sentinel Token Loss Handling: 모델의 학습 및 평가 과정에서는 센티널 토큰에 대한 손실이 모델에 반영되지 않도록 해야 합니다.
- 예를 들어, input = 'The <extra_id_0> walks in <extra_id_1> park'  labels = '<extra_id_0> cute dog <extra_id_1> the <extra_id_2>'이라면 모델은 <extra_id>를 예측하게 됩니다.
- 그러나, 이 <extra_id>는 '정답'이라기 보다는 빈칸에 대한 신호이기 때문에 <extra_id>에 대한 loss가 계산되는 것을 방지해야 합니다. 그렇지 않으면,  <extra_id_0> 다음에는 cute가 온다는 걸 학습해버리기 때문입니다.
- 그래서 <extra_id>에 대한 loss를 계산하지 않도록 하기 위해 이 센티널 토큰들을 pad_idx로 대체하였습니다. <code>T5_Slim_Attention/run_pretraining.py</code>의 <code>train</code> 및 <code>eval</code> 함수


- 배치 사이즈를 64로 사용할 때, 에폭당 스텝 수는 1,670이며 40 에폭 동안 학습할 경우, 총 학습 스텝 수는 66,800입니다. 이 중 10%를 웜업 스텝으로 설정하였습니다. <code>T5_Slim_Attention/config.py</code>
- 논문처럼 Adafactor optimizer와 inverse square root learning rate schedule을 사용했습니다. <code>T5_Slim_Attention/run_pretraining.py</code>


위와 같은 방법론에 따라 사전학습을 진행했을 때, train loss와 valid loss는 다음과 같았습니다. <code>T5_Slim_Attention/run_pretraining.ipynb</code>
<div align="center">
  <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/f687b12f-bc84-4f97-b48f-7d3bc90a1b3a" />
</div>


### 2.2 slim attention
빠른 추론 속도를 위해 KV-Cache를 사용한다면, 추론을 수행할 때 K/V에 대한 계산 결과를 메모리에 저장하기 때문에 아래 그림처럼 모델 파라미터와 KV-Cache가 GPU 메모리를 대부분 차지하게 됩니다.
<div align="center">
  <img width="200" height="150" alt="image" src="https://github.com/user-attachments/assets/1e12025b-63f7-4c6f-bb49-e4e7a65d79ac" />
</div>
- 위의 그림은 엔비디아 A100 40GB에서 13B 모델의 GPU 메모리 사용(전체 메모리 중 모델 파라미터가 약 65%를, KV-Cache가 30% 이상을 차지)을 시각화한 것입니다.

slim attention(https://arxiv.org/abs/2503.05840)을 이용하면, KV-Cache가 차지하는 크기를 절반으로 줄이면서도 어텐션 연산을 수행할 수 있습니다.

KV-Cache가 차지하는 크기가 절반을 줄어드는 이유는 V-Cache를 사용하지 않기 때문입니다. 

이것이 가능한 이유는 K로부터 V를 계산하기 때문입니다. 즉, 추론 과정에서 V-Cache를 사용하지 않고 K-Cache만 사용합니다. 단, 가중치 행렬이 정방행렬이어야 한다는 조건이 붙습니다.

입력 \(X\)에 대해  $K = XW_k, V = XW_v$로 계산됩니다. 만약 $W_k, W_v \in \mathbb{R}^{d \times d}$ 가 정방행렬이라면, 행렬식이 0이 아니라는 가정 하에 역행렬이 존재하기 때문에

$K = XW_k$의 양변에 $W^{-1}_k$를 곱해 $X = KW_k^{-1}$로 나타낼 수 있습니다. 

$X = KW_k^{-1}$를 $V = XW_v$에 대입하면, $V = XW_v = KW_k^{-1}W_v = KW_{kv}$가 성립합니다. 

여기서 $W_{kv} = W_k^{-1}W_v$를 추론 전에 미리 계산해두면, 추론 과정에서는 K만 캐싱해두고 필요할 때마다 $V = KW_{kv}$를 계산할 수 있습니다.

이것이 바로 slim attention 논문에서 주장하는 'K-Cache is all you need'이며, V-Cache를 사용하지 않기 때문에 기존 KV-Cache 메모리 사용량의 절반을 줄일 수 있습니다. <code>T5_Slim_Attention/slim_attention_and_relative_position_bias.py</code>

<div align="center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/867c82aa-3781-4ffa-a1d2-108b00b17256" />
</div>


## 3. fine-tuning
T5 연구에서는 사전학습된 모델을 다양한 다운스트림 태스크에 파인튜닝하는 방법을 비교 실험하였으며, 특히 어댑터 레이어를 활용할 경우 합리적인 성능을 유지할 수 있음을 확인하였습니다. 이 실험 결과에 착안하여 QLoRA를 사용했습니다. 

<div align="center">
  <img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/0be484dc-dd7b-43ce-97da-3cedd86672bf" />
</div>

### 3.1 qlora
QLoRA는 간단히 설명하면, 양자화된 선형 레이어를 사용하는 LoRA입니다. 

<div align="center">
  <img width="285" height="250" alt="image" src="https://github.com/user-attachments/assets/6f80cbc3-6160-405f-ac6a-f92cba474f09" />
</div> 

사전학습 모델의 가중치(위 그림의 왼쪽 가중치 행렬)는 NF4로 양자화되며 LoRA 가중치 A, B(위 그림의 오른쪽 가중치 행렬)는 BF16으로 계산됩니다. 이를 구현하기 위해 bitsandbytes 라이브러리를 이용하였습니다. 기본적인 LoRA에 대한 구현: <code>QLoRA/lora.py</code>, QLoRA에 대한 구현: <code>QLoRA/lora_4bit.py</code>

### 3.2 classification head fine-tuning
BERT에서는 classification task를 위해  모델에 classification head layer를 추가하고, <CLS> 토큰의 최종 은닉 벡터를 통과시킵니다. 

<div align="center">
  <img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/148239b3-74d5-42ff-a4d4-9261c199a3b7" />
</div>

<code>QLoRA/run_finetuning.py</code>는 이 방식을 full fine-tuning으로 구현한 코드입니다. 

### 3.3 text-to-text fine-tuning

<code>QLoRA/run_text_to_text_finetuning.py</code>는 기존 T5 모델의 가중치 파라미터를 동결하고, QLoRA 어댑터만 학습하도록 구현한 코드입니다. 

또한, T5의 text-to-text 접근법을 이용하였습니다. 구체적으로, <code>"sentiment: 뉴스 내용"</code>이라는 입력 텍스트를 제공하면, T5의 lm_head layer를 통해 <code>"Bearish"</code>, <code>"Neutral"</code>, <code>"Bullish"</code> 중 하나를 텍스트로 생성합니다. 

이때, 모델의 출력과 정답 텍스트가 정확히 일치(문자열 일치)하는 경우에만 정답으로 처리하였습니다. 

<div align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/0e8d3bf9-7ab2-4bb2-b66b-49e766000a8d" />
</div>

### 보완할 점
The Depth-to-Width Interplay in Self-Attention(https://arxiv.org/abs/2006.12467)에 따르면, 모델의 성능은 모델의 aspect ratio 변화에 비교적 강건하다.

본 프로젝트에서는 small model을 만들기 위해 6 layers를 사용했지만, 줄어든 레이어 수로 인한 성능 저하를 고려하지 않았다. 이를 보완하기 위해 상대적으로 넓은 은닉 차원(d_model)을 사용할 필요가 있다. 
