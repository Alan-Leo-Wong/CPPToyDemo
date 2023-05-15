#pragma once

namespace igl {
// Return the current time in seconds since program start
//
// Example:
//    const auto & tictoc = []()
//    {
//      static double t_start = igl::get_seconds();
//      double diff = igl::get_seconds()-t_start;
//      t_start += diff;
//      return diff;
//    };
//    tictoc();
//    ... // part 1
//    cout<<"part 1: "<<tictoc()<<endl;
//    ... // part 2
//    cout<<"part 2: "<<tictoc()<<endl;
//    ... // etc
inline double get_seconds();

} // namespace igl