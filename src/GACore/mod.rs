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
    string: [[[u8; 4]; 3]; 4],
    fitness: f64,
}

/*
  * Separation Genome structure
  */
#[derive(Debug, Copy, Clone)]
struct SegGenome {
    string: [[[u8; 10]; 6]; 10],
    fitness: f64,
}