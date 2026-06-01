# Build the Beamer slides. Requires a TeX distribution with latexmk
# (ships with MacTeX / TeX Live).
#
#   make        # build main.pdf
#   make watch  # rebuild automatically on file changes
#   make clean  # remove aux files (keeps main.pdf)
#   make purge  # remove aux files AND main.pdf

MAIN = main

.PHONY: all watch clean purge

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

purge:
	latexmk -C $(MAIN).tex
