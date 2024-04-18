#!/bin/bash
##Run this only if code changes
cargo build -r
for (( i=0; i<=20;i++))
do
./target/release/swarm_aggregation_ga -b brid -e th -g 500 -p 500 --gran 10 -m 0.006 '-k(15,3,17,90)' '-k(15,3,17,60)' '-k(15,3,17,30)' '-w(0.00,2.00,6.00,2.00)' -t $i --output-path ./output/ &
wait $!
done