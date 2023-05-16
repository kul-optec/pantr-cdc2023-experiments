#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/..

set -ex

mkdir -p images
for pdf in new-benchmarks-paper/output/*.pdf; do
    f=$(basename $pdf)
    gs -dNoOutputFonts -sDEVICE=pdfwrite -o /tmp/$f $pdf
    inkscape /tmp/$f -z -l=images/$f.svg
done
