PYTHON ?= python3
EXAMPLE := examples/interference

.PHONY: convert render inspect test

convert:
	PYTHONPATH=src $(PYTHON) -m hybrid_vectorizer convert \
		$(EXAMPLE)/raster.png \
		-o build/auto.svg \
		--report build/auto.report.json \
		--preview build/auto.png

render:
	PYTHONPATH=src $(PYTHON) -m hybrid_vectorizer render \
		$(EXAMPLE)/spec.json \
		-o $(EXAMPLE)/raster_hybrid.svg \
		--outlined $(EXAMPLE)/raster_hybrid_outlined.svg \
		--preview $(EXAMPLE)/raster_hybrid_preview.png

inspect:
	PYTHONPATH=src $(PYTHON) -m hybrid_vectorizer inspect \
		$(EXAMPLE)/spec.json \
		--output-dir build/ocr

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
