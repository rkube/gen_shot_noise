#include <iostream>
#include <string>
#include <map>
#include "include/gen_shot_noise.h"
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

    // Write the configuration file as attributes
    map<dist_t, std::string> my_map{
        {dist_t::expon_t, string("exponential")},
        {dist_t::normal_t, string("normal")},
        {dist_t::uniform_t, string("uniform")}};

    int att_val_int{0};         // Store attribute value for integer values
    double att_val_double{0.0}; // Store attribute value for double values
    vector<double> att_vec_double;
    const hsize_t att_vec_dims{2};
    H5std_string att_str;       // Store attribute value for string values

    // Create new string datatype for attribute
    H5::StrType strdatatype(H5::PredType::C_S1, 32); // of length 32 characters

    H5::DataSpace attrib_space(H5S_SCALAR);         // Dataspace for scalar attributes
    H5::DataSpace attrib_space2(1, &att_vec_dims);  // Dataspace for 2 element vectors
    H5::Attribute att = dataset -> createAttribute("num_bursts", H5::PredType::NATIVE_INT, attrib_space);
    att_val_int = sn_config.get_num_bursts();
    att.write(H5::PredType::NATIVE_INT, &att_val_int);
    
    att = dataset -> createAttribute("dt", H5::PredType::NATIVE_DOUBLE, attrib_space);
    att_val_double = sn_config.get_dt();
    att.write(H5::PredType::NATIVE_DOUBLE, &att_val_double);

    att = dataset -> createAttribute("Tend", H5::PredType::NATIVE_DOUBLE, attrib_space);
    att_val_double = sn_config.get_Tend();
    att.write(H5::PredType::NATIVE_DOUBLE, &att_val_double);

    att = dataset -> createAttribute("Lpar", H5::PredType::NATIVE_DOUBLE, attrib_space);
    att_val_double = sn_config.get_Lpar();
    att.write(H5::PredType::NATIVE_DOUBLE, &att_val_double);

    att = dataset -> createAttribute("Cs", H5::PredType::NATIVE_DOUBLE, attrib_space);
    att_val_double = sn_config.get_Cs();
    att.write(H5::PredType::NATIVE_DOUBLE, &att_val_double);

    att = dataset -> createAttribute("Lsol", H5::PredType::NATIVE_DOUBLE, attrib_space);
    att_val_double = sn_config.get_Lsol();
    att.write(H5::PredType::NATIVE_DOUBLE, &att_val_double);

    att = dataset -> createAttribute("Amplitude distribution", strdatatype, attrib_space);
    att_str = my_map.at(sn_config.get_dist_amp_type());
    att.write(strdatatype, att_str);

    att_vec_double = sn_config.get_dist_amp_params();
    att = dataset -> createAttribute("Amplitude distribution parameters", H5::PredType::NATIVE_DOUBLE, attrib_space2);
    att.write(H5::PredType::NATIVE_DOUBLE, att_vec_double.data());

    att = dataset -> createAttribute("Arrival time distribution", strdatatype, attrib_space);
    att_str = my_map.at(sn_config.get_dist_arrival_type());
    att.write(strdatatype, att_str);

    att_vec_double = sn_config.get_dist_arrival_params();
    att = dataset -> createAttribute("Arrival time distribution parameters", H5::PredType::NATIVE_DOUBLE, attrib_space2);
    att.write(H5::PredType::NATIVE_DOUBLE, att_vec_double.data());

    att = dataset -> createAttribute("Length distribution", strdatatype, attrib_space);
    att_str = my_map.at(sn_config.get_dist_length_type());
    att.write(strdatatype, att_str);

    att_vec_double = sn_config.get_dist_length_params();
    att = dataset -> createAttribute("Length distribution parameters", H5::PredType::NATIVE_DOUBLE, attrib_space2);
    att.write(H5::PredType::NATIVE_DOUBLE, att_vec_double.data());

    att = dataset -> createAttribute("Velocity distribution", strdatatype, attrib_space);
    att_str = my_map.at(sn_config.get_dist_vrad_type());
    att.write(strdatatype, att_str);

    att_vec_double = sn_config.get_dist_vrad_params();
    att = dataset -> createAttribute("Velocity distribution parameters", H5::PredType::NATIVE_DOUBLE, attrib_space2);
    att.write(H5::PredType::NATIVE_DOUBLE, att_vec_double.data());


}


void output_t :: write_to_tlev(size_t xi_idx, double* data)
{
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
