#! /bin/sh

wget https://www.dropbox.com/s/hmi7i6jcrkkwhxk/65700.h5 | tr -d '\r'
wget https://www.dropbox.com/s/ssy2rb7cp5t3r1u/64753.h5 | tr -d '\r'
python3 ensamble_test.py $1 $2
