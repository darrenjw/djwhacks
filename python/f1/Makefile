# Makefile

YEARS=2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024

PARQUET=$(patsubst %,results-%.parquet,$(YEARS))

FORCE:
	make all-results.parquet

results-%.parquet: results.py
	python results.py $(word 2, $(subst -, ,$(word 1, $(subst ., ,$@))))

all-results.parquet: $(PARQUET)
	python combine.py $(PARQUET)



# eof

