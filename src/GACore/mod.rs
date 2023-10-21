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

#[derive(Debug, Copy, Clone)]
enum DiversThresh {
  /// Initial state when diversity is high and mutation rate is standard
  INIT,
  /// Low diversity state and hence mutation rate is x2
  LOWER_HIT,
}