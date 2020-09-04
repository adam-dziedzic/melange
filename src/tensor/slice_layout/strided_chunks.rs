use super::SliceLayout;

pub struct StridedChunks<'a, 'b, T> {
    counter: Vec<usize>,
    step_sizes: Vec<usize>,
    dead: bool,
    chunk_size: usize,
    layout: &'b SliceLayout<'a, T>,
}

impl<'a, 'b, T> StridedChunks<'a, 'b, T> {
    pub fn new(layout: &'a SliceLayout<'b, T>, chunk_size: usize) -> Self {
        let mut step_sizes = layout.shape.clone();
        let mut chunk_size = chunk_size;

        step_sizes
            .iter_mut()
            .rev()
            .for_each(|x| {
                if chunk_size < *x {
                    *x = 1
                } else {
                    chunk_size /= *x
                }
            });

        StridedChunks {
            counter: vec![0; layout.num_elements],
            dead: false,
            step_sizes,
            chunk_size,
            layout,
        }
    }
}

impl<'a, 'b, T> Iterator for StridedChunks<'a, 'b, T>
where
    'b: 'a,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.dead {
            return None;
        }
        let chunk = self.layout.chunk_at(&self.counter, self.chunk_size);

        for ((digit, step_size), bound) in self
            .counter
            .iter_mut()
            .zip(self.step_sizes.iter())
            .zip(self.layout.shape.iter())
            .rev()
        {
            if *digit + step_size >= *bound {
                *digit = 0;
            } else {
                *digit += step_size;
                return Some(chunk);
            }
        }

        self.dead = true;
        Some(chunk)
    }
}
