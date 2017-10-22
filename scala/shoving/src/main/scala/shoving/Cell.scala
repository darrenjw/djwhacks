package shoving

import breeze.stats.distributions.{ Uniform, Gaussian }

/**
 * Cell agent representing the state and behaviour of a single cell
 *
 * @constructor Make a cell with given attributes
 * @param x the x coordinate of the spatial position of the cell
 * @param y the y coordinate of the spatial position of the cell
 * @param z the z coordinate of the spatial position of the cell
 * @param s the size (volume) of the cell
 * @param a the age of the cell (in time steps)
 */
case class Cell(x: Double, y: Double, z: Double, s: Double, a: Double) {

  /**
   * Random Brownian spatial drift of the cell over a single time step
   *
   * @return New cell at updated spatial position
   */
  def drift: Cell = {
    val dc = 0.01
    Cell(x + Gaussian(0, dc).draw, y + Gaussian(0, dc).draw, z + Gaussian(0, dc).draw, s, a)
  }

  /**
   * Random growth of a cell over a single time step
   *
   * @return New cell with updated size
   */
  def grow: Cell = {
    Cell(x, y, z, s + Uniform(0.001, 0.01).draw, a)
  }

  /**
   * Age the cell by one time step
   *
   * @return New cell with updated age
   */
  def age: Cell = {
    Cell(x, y, z, s, a + 1)
  }

  /**
   * Divide the cell in two if the size exceeds 2
   *
   * @return List of either 1 or 2 cells depending on whether the cell divides
   */
  def divide: List[Cell] = {
    if (s < 2.0)
      List(this)
    else {
      val jit = Gaussian(0, 0.1).sample(3)
      List(Cell(x - jit(0), y - jit(1), z - jit(2), 1.0, a / 4), Cell(x + jit(0), y + jit(1), z + jit(2), 1.0, 0))
    }
  }

  /**
   * Randomly kill the cell
   *
   * @return List of either 0 or 1 cells depending on whether the cell dies
   */
  def die: List[Cell] = {
    if (Uniform(0, 1).draw < 0.0005 * math.log(a + 1))
      List()
    else
      List(this)
  }

  /**
   * Calculate the force of this cell on another cell, c.
   * Although this is computed (very roughly) according to a very simple-minded force-field,
   * it is actually used in the code as a spatial displacement to be applied over a time step.
   *
   * @param c a cell to be compared to this cell
   * @return The force that this cell exerts on cell c
   */
  def force(c: Cell): Force = {
    val dx = x - c.x
    val dy = y - c.y
    val dz = z - c.z
    val dr = math.sqrt(dx * dx + dy * dy + dz * dz)
    val td = math.pow(s, 1.0 / 3) + math.pow(c.s, 1.0 / 3)
    if (dr > 2 * td)
      Force(0, 0, 0)
    else {
      //val f = 0.01 * (dr - td) * (dr - 2 * td)
      //val f = if (dr < td) 0.02 else -0.001
      val f = 0.01 * (if (dr < td) (10.0 - 5.0 * dr / td) else (-2.0 + dr / td))
      Force(f * dx / dr, f * dy / dr, f * dz / dr)
    }
  }

  /**
   * Shift a cell according to a given force, f.
   *
   * @param f a force (spatial displacement) to be applied to a cell
   * @return The cell in the new spatial position
   */
  def shift(f: Force): Cell = Cell(x - f.dx, y - f.dy, z - f.dz, s, a)

  /**
   *  Rotate a cell around the y axis.
   *  Used to provide a rotating view of the cell population
   *
   *  @param th the angle in radians to rotate by (in a single time step)
   *  @return New cell in the rotated position
   */
  def rotate(th: Double): Cell = Cell(x * math.cos(th) - z * math.sin(th), y, x * math.sin(th) + z * math.cos(th), s, a)

}

case class Force(dx: Double, dy: Double, dz: Double) {
  def add(f: Force): Force = Force(dx + f.dx, dy + f.dy, dz + f.dz)
}
