# 종이 상자 정상/불량 검사

## 데이터 구조

`archive/intact/top`, `archive/intact/side`의 정상 이미지를 학습하고, `archive/damaged/top`, `archive/damaged/side`의 불량 이미지를 판정 임계값 보정에 사용합니다. 이미지 확장자는 jpg, jpeg, png, bmp, webp를 지원합니다.

## 설치 및 학습

```powershell
python -m pip install -r requirements.txt
python train_autoencoder.py --data-dir archive --epochs 30 --batch-size 8 --img-size 256 --out-dir model_out
```

학습 결과로 `model_out/ae_final.pth`, `metadata.json`, `threshold.json`이 만들어집니다. 제공된 불량 사진을 이용해 정상/불량 점수의 균형 정확도가 가장 높아지도록 임계값을 보정합니다. 불량 사진이 없으면 정상 검증 이미지 점수의 99번째 백분위수를 사용합니다.

## 웹 앱

```powershell
streamlit run streamlit_app.py
```

앱에서 기본 모델과 임계값 경로를 그대로 사용하고 모델을 로드한 뒤 검사할 이미지를 올립니다. GPU가 감지되지 않으면 CPU만 선택할 수 있습니다.

## 폴더 추론

```powershell
python infer_Version2.py --model-dir model_out --input-folder new_images --out-csv results.csv
```

CSV에는 이미지 경로, 재구성 MSE, 판정, 임계값이 기록됩니다. 현재 오토인코더 구조는 256×256 입력을 사용합니다.
