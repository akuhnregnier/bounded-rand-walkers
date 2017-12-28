// Copyright 2000 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

#ifndef POLYGON_INCLUSION_H
#define POLYGON_INCLUSION_H

#include <vector>
#include <cstddef>
#include "common.h"

// a Point is defined by its coordinates {int x, y;}
//===================================================================
typedef struct {double x, y;} Point;
typedef std::vector<Point> pvect;


// isLeft(): tests if a point is Left|On|Right of an infinite line.
//    Input:  three points P0, P1, and P2
//    Return: >0 for P2 left of the line through P0 and P1
//            =0 for P2  on the line
//            <0 for P2  right of the line
//    See: Algorithm 1 "Area of Triangles and Polygons"
inline int
isLeft( Point P0, Point P1, Point P2 )
{
    return ( (P1.x - P0.x) * (P2.y - P0.y)
            - (P2.x -  P0.x) * (P1.y - P0.y) );
}
//===================================================================


// cn_PnPoly(): crossing number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  0 = outside, 1 = inside
// This code is patterned after [Franklin, 2000]
int
cn_PnPoly( Point P, Point* V, int n )
{
    int    cn = 0;    // the  crossing number counter

    // loop through all edges of the polygon
    for (int i=0; i<n; i++) {    // edge from V[i]  to V[i+1]
       if (((V[i].y <= P.y) && (V[i+1].y > P.y))     // an upward crossing
        || ((V[i].y > P.y) && (V[i+1].y <=  P.y))) { // a downward crossing
            // compute  the actual edge-ray intersect x-coordinate
            float vt = (float)(P.y  - V[i].y) / (V[i+1].y - V[i].y);
            if (P.x <  V[i].x + vt * (V[i+1].x - V[i].x)) // P.x < intersect
                 ++cn;   // a valid crossing of y=P.y right of P.x
        }
    }
    return (cn&1);    // 0 if even (out), and 1 if  odd (in)

}
//===================================================================


// wn_PnPoly(): winding number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  wn = the winding number (=0 only when P is outside)
int
wn_PnPoly( Point P, Point* V, int n )
{
    int    wn = 0;    // the  winding number counter

    // loop through all edges of the polygon
    for (int i=0; i<n; i++) {   // edge from V[i] to  V[i+1]
        if (V[i].y <= P.y) {          // start y <= P.y
            if (V[i+1].y  > P.y)      // an upward crossing
                 if (isLeft( V[i], V[i+1], P) > 0)  // P left of  edge
                     ++wn;            // have  a valid up intersect
        }
        else {                        // start y > P.y (no test needed)
            if (V[i+1].y  <= P.y)     // a downward crossing
                 if (isLeft( V[i], V[i+1], P) < 0)  // P right of  edge
                     --wn;            // have  a valid down intersect
        }
    }
    return wn;
}
//===================================================================

/* IC - 28/12/2017 */


template <class T>
pvect make_points(const T& arr) {
    pvect points;
    for (size_t i(0); i<arr.size(); ++i) {
        points.push_back(Point());
        points[i].x = arr[i][0];
        points[i].y = arr[i][1];
    }
    return points;
}


bool in_shape(const dvect& P_in, const std::vector<dvect>& points) {
    /* P_in is a `dvect` containing the 2 coordinates of the point to test.
     * points is a vector of `dvect`, containing the points of the polygon in
     *      clockwise order.
     * vertices.
     */
    Point P;
    P.x = P_in[0];
    P.y = P_in[1];
    return cn_PnPoly(P, &make_points(points)[0], points.size() - 1);
}


void test_polygon() {
    vect_dvect point_vect;
    point_vect.push_back(dvect {0, 0});
    point_vect.push_back(dvect {0, 1});
    point_vect.push_back(dvect {1, 0});
    point_vect.push_back(dvect {0, 0});
    auto points = make_points(point_vect);

    print("testing polygon inclusion");
    Point test;
    test.x = 0.5;
    test.y = 0.3;

    for (size_t i(0); i<points.size(); ++i) {
        printf("x=%g, y=%g\n", ((&points)[0])[i].x, ((&points)[0])[i].y); // this works as intended
    }
    print("test point");
    printf("x=%g, y=%g\n", test.x, test.y); // this works as intended

    print(cn_PnPoly(test, &make_points(point_vect)[0], points.size() - 1));
    print(in_shape(dvect {test.x, test.y}, point_vect));
}


#endif
