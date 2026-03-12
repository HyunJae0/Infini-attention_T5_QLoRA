# US Stock Market News Sentiment Classification

## 1. 프로젝트 목표
프로젝트의 목표는 미국 증시 관련 뉴스 문장을 입력받아, 해당 뉴스가 시장 관점에서 bearish, neutral, bullish 가운데 어느 방향성에 해당하는지를 예측하는 domain-specific small language model을 구축하는 것이다. 
 
프로젝트에서 모델 학습에 대한 가설은 small scale unlabeld text corpus로 pre-training한 다음, 동일한 도메인의 task 대해 fine-tuning을 수행하여 합리적인 성능을 달성할 수 있다는 것이다. 

약 54M의 pre-training data로 학습한 모델이 downstream task에서 약 83%의 정확도 달성하였다. 이는 일반적인 말뭉치만으로는 포착하기 어려운 실적 발표 표현, 가격 변동 서술, 거시경제 이벤트, 금융 전문 용어 등에 따른 패턴을 충분히 모델 파라미터에 내재화함으로써, downstream task에서 합리적인 수준의 성능을 달성할 수 있음을 보여준다. 

---

## 2. 데이터셋 선정
