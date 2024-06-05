# SOPS-SwarmGA
This is a research project in Swarm robotics using Genetic Algorithms as the backbone, to search for local controllers to produce global collective behaviors.  The base model used is the SOPS model from collective research lab at ASU.

## RUST setup
**Make sure you have rust installed\***
To install latest rust compiler run command:<br>
`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`<br>
Then to compile and run the simulation run command:<br>
`cargo run`<br><br>

## Basic Repository Structure

The `main.rs` file houses the execution loop to run a full scale Genetic Algorithm or a Standalone evaluation of a genome. The RUST modules are organized based on the 2 main components of the code-base the SOPS environment and Genetic Algorithm housed respectively in the `SOPSCore` and `GACore` folder.<br>
The modules have a main module file which contain all the code shared within the respective module files. Separate files are maintained for the Separation behavior which is implemented as an extension of the Aggregation behavior.<br>
**Please use a similar file format for making any extensions of new behaviors\*** <br>
The `utils` module contains all the complementary helper functions required for main experiment tasks. There is another folder `misc_scripts` folder which contains python scripts used to generate all the auxiliary data and graphs for/from the experiments.

## Running the experiments

Pass options to run the experiments<br>
`cargo run --package swarm_aggregation_ga --bin swarm_aggregation_ga -- --behavior <BEHAVIOR> --exp <EXPERIMENT_TYPE> --ks <PARTICLE_SIZES> ...(other options)`<br>

Printing out the options: `cargo run --package swarm_aggregation_ga --bin swarm_aggregation_ga -- --help` 

### Options:<br>
```
-b, --behavior <BEHAVIOR>
          Behavior to run

          Possible values:
          - agg: Aggregation
          - sep: Separation
          - loco: Locomotion
          - coat: Coating

  -e, --exp <EXPERIMENT_TYPE>
          Type of Experiment to run

          Possible values:
          - ga: Full scale genetic algorithm run
          - gm: Stand-alone single genome run
          - th: Stand-alone single theory given solution run

  -g, --gen <MAX_GENERATIONS>
          Maximum no. of generations to run the Genetic Algorithm Experiment
          
          [default: 0]

  -p, --pop <POPULATION>
          No. of genomes in the population for a Genetic Algorithm Experiment
          
          [default: 0]

      --gran <GRANULARITY>
          Genome representation granularity
          
          [default: 20]

  -m, --mut <MUTATION_RATE>
          Mutation rate per gene
          
          [default: 0.08]

      --eli <ELITIST_COUNT>
          Maximum no. of elite genomes to preserve in a generation
          
          [default: 0]

  -k, --ks <PARTICLE_SIZES>
          Particle Sizes to run on (eg. use multiple -k<String"(<u64>,<u64>)"> arguments to specify multiple sizes)

  -s, --seed <SEEDS>
          Seed values to run experiments with, for reproducible trials (eg. use multiple -s<u64> arguments to specify seeded trials)

      --path <FILE>
          File to read genome value from

  -v, --verbose
          Specify if execution description is written to the output

      --snaps
          Specify if snapshots of the experiment run are written to the output (only valid for stand-alone experiments)

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
