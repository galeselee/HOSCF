if [ -d "./build" ]; then
    rm -rf build
fi
mkdir build

cd build
cmake ..
make -j19

mv rank-one ..
