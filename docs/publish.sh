#!/bin/bash

echo "Building Documentation..."
cd gitbook-src/
gitbook build
cp -r _book/* ../
cd ../
echo "Done"
