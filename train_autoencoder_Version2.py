간단 사용법

1) 데이터 준비
- 데이터 루트 예: data/top/, data/side/ 에 정상 이미지들을 넣습니다.
- (선택) 불량 이미지가 있을 경우 별도 폴더에 두고 추후 평가용으로 사용 가능.

2) 설치
- 가상환경 생성 후:
  pip install -r requirements.txt

3) 학습
  python train_autoencoder.py --data-dir data --epochs 30 --batch-size 32 --img-size 256 --out-dir model_out

4) 추론
  python infer.py --model-dir model_out --input-folder new_images --out-csv results.csv

결과: results.csv 에 각 이미지의 MSE와 'normal'/'defect' 판정이 저장됩니다.

팁:
- threshold 설정은 metadata.json 의 percentile로 정해집니다. 더 보수적으로 하려면 percentile 값을 높이세요(예: 99).
- 불량 부위 국소화가 필요하면 patch-wise reconstruction MSE를 계산해 heatmap으로 시각화할 수 있습니다.