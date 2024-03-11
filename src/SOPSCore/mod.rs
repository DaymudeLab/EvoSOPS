pub mod bridging;

#[derive(Eq, PartialEq, Hash)]
struct Particle {
    x: u8,
    y: u8,
    onland: bool,
}
