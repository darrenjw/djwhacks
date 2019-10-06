/*
a-star.scala

A*-search puzzle for FP North East

https://github.com/FP-North-East/atomic-space-helicopters/

https://en.wikipedia.org/wiki/A*_search_algorithm

 */

// Reasonably generic A* path finder
object AStar {

  import simulacrum._
  @typeclass trait GraphNode[N] {
    def neighbours(n: N): List[N]
  }
  import GraphNode.ops._

  case class State[N](
      cameFrom: Map[N, N],
      closedSet: Set[N],
      openSet: Map[N, Double],
      gScore: Map[N, Double]
  )

  def aStar[N: GraphNode](
      start: N,
      target: N,
      h: N => Double
  ): (List[N], State[N]) = {
    val cameFrom  = Map[N, N]()
    val closedSet = Set[N]()
    val openSet   = Map[N, Double](start -> h(start))
    val gScore    = Map[N, Double](start -> 0.0).withDefaultValue(Double.PositiveInfinity)
    val solution  = findPath(State(cameFrom, closedSet, openSet, gScore), h, target)
    val path      = reconstuctPath(solution.cameFrom, List(target))
    (path, solution)
  }

  @annotation.tailrec
  def reconstuctPath[N](cameFrom: Map[N, N], current: List[N]): List[N] = {
    val top = current.head
    if (cameFrom.contains(top))
      reconstuctPath(cameFrom, cameFrom(top) :: current)
    else
      current
  }

  @annotation.tailrec
  def findPath[N: GraphNode](
      state: State[N],
      h: N => Double,
      target: N
  ): State[N] = {
    val (current, _) = state.openSet.reduce((kv1, kv2) => if (kv2._2 < kv1._2) kv2 else kv1)
    if (current == target) state
    else {
      val state1 = State(
        state.cameFrom,
        state.closedSet + current,
        state.openSet.removed(current),
        state.gScore
      )
      val newState = current.neighbours.foldLeft(state1)((st, ne) => {
        if (st.closedSet.contains(ne)) st
        else {
          val tentativeGScore = st.gScore(current) + 1.0
          // Distance between neighbours is 1 for the example,
          //  but obviously won't be in general.
          // Could pass in a distance function, d(), just like h().
          if (tentativeGScore >= st.gScore(ne)) st
          else {
            State(
              st.cameFrom.updated(ne, current),
              st.closedSet - ne, // Correct to remove the neighbour?
              st.openSet + ((ne, tentativeGScore + h(ne))),
              st.gScore.updated(ne, tentativeGScore)
            )
          }
        }
      })
      findPath(newState, h, target)
    }
  }

}
// eof
