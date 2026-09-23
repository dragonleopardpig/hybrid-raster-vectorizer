PYTHON ?= python3
EXAMPLE := examples/interference

.PHONY: render inspect test

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
