#include <iostream>
#include "include/output.h"

using namespace std;

output_t :: output_t(config_t config_) : sn_config(config_),
    file(sn_config.get_outfile_name().data(), H5F_ACC_TRUNC),
    dset_dims{sn_config.get_num_xi(), sn_config.get_nelem()},
    file_space(2, dset_dims)
{
    // We do chunked access to the dataset, see
    // https://www.hdfgroup.org/HDF5/doc/Advanced/Chunking/index.html   
    // https://www.hdfgroup.org/HDF5/doc/cpplus_RM/writedata_8cpp-example.html

    // Create a dataset creation property list, use chunking
    constexpr double fill_val{-1.0};
    prop_list.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_val);

    // Create dataset and write to file
    dataset = new H5::DataSet(file.createDataSet("shotnoise", H5::PredType::NATIVE_DOUBLE, file_space, prop_list));
    cout << "done" << endl;
}


void output_t :: write_to_tlev(size_t xi_idx, double* data)
{
    cout << "writing for xi =  " << xi_idx << endl;

    // We need a reference to this guy later on
    const hsize_t nelem{sn_config.get_nelem()};
    //Select hyperslab in the dataset of the output file 
    hsize_t start[2] {xi_idx, 0};   // Start of hyperslab
    hsize_t stride[2] {1, 1};       // Stride of hyperslab, do not worry about it.
    hsize_t count[2] {1, 1};        // We choose only one block,
    hsize_t block[2] {1, nelem};    // which extends along the enire 2nd axis
    file_space.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

    // Create a dataspace for the data in memory
    H5::DataSpace memory_space(1, &nelem);
    start[0] = 0;
    stride[0] = 1;
    count[0] = 1;
    block[0] = nelem;
    memory_space.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

    // Writing to file 
    dataset -> write(data, H5::PredType::NATIVE_DOUBLE, memory_space, file_space);
}


output_t :: ~output_t()
{
    file.close();
    delete dataset;
}


// End of file output.cpp
