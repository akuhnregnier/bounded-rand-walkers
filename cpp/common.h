#ifndef COMMON_H
#define COMMON_H


typedef std::vector<Point> pvect;
typedef std::vector<double> dvect;
typedef xt::xarray<double> dxarray;


struct pdf_data {
    dvect centre;
    double width;
};


#endif
