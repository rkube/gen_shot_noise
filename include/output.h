#ifndef output_h
#define output_h

#include "config.h"
#include <H5Cpp.h>

class output_t
{
    public:
        output_t(config_t);
        ~output_t();
        void write_to_tlev(size_t, double*);

    private:
        const config_t sn_config;
        H5::H5File file;
        hsize_t dset_dims[2];
        H5::DSetCreatPropList prop_list;
        H5::DataSpace file_space;
        H5::DataSet* dataset{nullptr};
};


#endif // output_h
