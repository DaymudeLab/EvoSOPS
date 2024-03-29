#!/bin/bash
##Run this only if code changes
cargo build -r
./target/release/swarm_aggregation_ga -b brid -e ga -g 1250 -p 600 --gran 10 -m 0.007 '-k(15,3,17,90)' '-w(1.00,5.00,3.00,1.50)' --path ./output/Brid_genome.log