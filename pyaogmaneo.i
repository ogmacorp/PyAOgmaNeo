// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// %begin %{
// #include <cmath>
// #include <iostream>
// %}
%module pyaogmaneo

%include "std_array.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "PyHelpers.h"
#include "PyHierarchy.h"
#include "PyImageEncoder.h"
%}

// Handle STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%ignore pyaon::PyStreamReader;
%ignore pyaon::PyStreamWriter;

%template(StdArray3i) std::array<int, 3>;
%template(StdVeci) std::vector<int>;
%template(StdVec2Di) std::vector<std::vector<int> >;
%template(StdVecf) std::vector<float>;
%template(StdVec2Df) std::vector<std::vector<float> >;
%template(StdVecuchar) std::vector<unsigned char>;
%template(StdVec2Duchar) std::vector<std::vector<unsigned char> >;
%template(StdVecInt3) std::vector<std::array<int, 3> >;
%template(StdVecIODesc) std::vector<pyaon::PyIODesc>;
%template(StdVecLayerDesc) std::vector<pyaon::PyLayerDesc>;
%template(StdVecImageEncoderVisibleLayerDesc) std::vector<pyaon::PyImageEncoderVisibleLayerDesc>;

%rename("%(strip:[Py])s") ""; // Remove Py prefix that was added to avoid naming collisions

%include "PyHelpers.h"
%include "PyHierarchy.h"
%include "PyImageEncoder.h"
