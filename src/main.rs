#[macro_use]
extern crate rustacuda;
extern crate rustacuda_core;

use rustacuda::{memory::DeviceBuffer, prelude::*};
use std::{error::Error, ffi, time::Instant};

const IMAGE_WIDTH: u32 = 8000;
const IMAGE_HEIGHT: u32 = 6000;
const X_OFFSET: i32 = (-(IMAGE_WIDTH as f32) / 3.0) as i32;
const Y_OFFSET: i32 = -(IMAGE_HEIGHT as i32) / 4;
const SCALE: f32 = 0.001;

fn main() -> Result<(), Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    println!("[info] Successfully initialized CUDA instance.");

    let device = Device::get_device(0)?;
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    println!("[info] Successfully created context.");

    let module = Module::load_from_file(&ffi::CString::new("./res/lib.ptx")?)?;
    println!("[info] Successfully loaded compute code.");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut px_lines: Vec<DeviceBuffer<u8>> = Vec::with_capacity(IMAGE_HEIGHT as usize);
    for _ in 0..px_lines.capacity() {
        px_lines.push(unsafe { DeviceBuffer::uninitialized(IMAGE_WIDTH as usize)? });
    }
    println!("[info] Successfully allocated device memory.");

    let timer = Instant::now();
    for (idx, line) in px_lines.iter_mut().enumerate() {
        unsafe {
            // errors out here
            launch!(module.render_line<<<16, IMAGE_WIDTH / 16, 0, stream>>>(line.as_device_ptr(), X_OFFSET, idx as i32 / 2 + Y_OFFSET, SCALE))?;
        }
    }

    stream.synchronize()?;

    let mut imgbuf = Vec::with_capacity(IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize);
    for line in px_lines {
        let mut host_line = [0; IMAGE_WIDTH as usize];
        line.copy_to(&mut host_line)?;
        imgbuf.append(&mut Vec::from(host_line));
    }
    println!("[info] Rendering finished, took {} seconds.", timer.elapsed().as_secs_f32());
    println!("[info] Saving to image.png...");
    image::save_buffer(
        "image.png",
        &imgbuf[..],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        image::ColorType::L8,
    )?;
    println!("[info] Successfully saved to image.png!");

    Ok(())
}
