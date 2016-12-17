#!/bin/sh
export PATH=$(pwd):$PATH
mkdir tmp
mmseqs createdb $1 DB
mmseqs cluster DB clu tmp --cluster-fragments
mmseqs createseqfiledb DB clu clu_seq
mmseqs createtsv DB DB clu $2
rm -r tmp
