# Build the ACL report. Requires a TeX distribution with latexmk
# (ships with MacTeX / TeX Live).
#
#   make        # build main.pdf (runs pdflatex + bibtex as needed)
#   make watch  # rebuild automatically on file changes
#   make clean  # remove aux files (keeps main.pdf)
#   make purge  # remove aux files AND main.pdf

MAIN = main

.PHONY: all watch clean purge

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex refs.bib
	latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -bibtex -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

purge:
	latexmk -C $(MAIN).tex
