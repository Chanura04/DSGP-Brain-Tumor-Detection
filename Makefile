# Run everything
all: data preprocess train evaluate

data:
	python src/data/download_data.py

preprocess:
	python src/data/preprocess.py

train:
	python src/models/train.py --config configs/train_config.yml

evaluate:
	python src/models/evaluate.py

clean:
	rm -rf data/processed/*
	rm -rf results/outputs/*
