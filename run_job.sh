#!/bin/bash
##Run this only if code changes
cargo build -r
for (( i=0; i<=20;i++))
do
./target/release/swarm_aggregation_ga -b brid -e th -g 200 -p 600 --gran 10 -m 0.006 "-k(20,3,17,30)" "-k(20,10,17,30)" "-k(20,7,23,30)" "-k(20,3,17,60)" "-k(20,17,17,60)" "-k(20,8,23,60)" "-k(20,5,17,90)" '-w(0.00,2.00,6.00,2.00)' -t $i --output-path ./output/ &
wait $!
done