pub mod base_ga;
pub mod seg_ga;

/*
 * Main file all the basic classes needed for GA to work
 *  */

 /*
  * Aggregagtion Genome structure (use as a basis to derive and create for other behaviors)
  */
#[derive(Debug, Copy, Clone)]
struct Genome {
    string: [u16; 6],
    fitness: f64,
}

/*
  * Separation Genome structure
  */
#[derive(Debug, Copy, Clone)]
struct SegGenome {
    string: [[[u16; 6]; 7]; 7],
    fitness: f64,
}