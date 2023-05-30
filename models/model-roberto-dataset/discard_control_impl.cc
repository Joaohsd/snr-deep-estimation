/* -*- c++ -*- */
/*
 * Copyright 2023 Inatel.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "discard_control_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace customModule {

using input_type = gr_complex;
using output_type = gr_complex;

discard_control::sptr discard_control::make(unsigned int num_qam)
{
    return gnuradio::make_block_sptr<discard_control_impl>(num_qam);
}


/*
 * The private constructor
 */
discard_control_impl::discard_control_impl(unsigned int num_qam)
    : gr::block("discard_control",
                gr::io_signature::make(
                    1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                gr::io_signature::make(
                    1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
    d_num_qam_symbols = num_qam;
    d_num_ctrl_symbols = num_qam%1024;
    d_num_data_symbols = num_qam - d_num_ctrl_symbols;

    d_last_statistics_time = std::chrono::system_clock::now();

    set_output_multiple(d_num_data_symbols);
    set_relative_rate(float(d_num_data_symbols)/float(d_num_qam_symbols));
}

/*
 * Our virtual destructor.
 */
discard_control_impl::~discard_control_impl() {}

void discard_control_impl::forecast(int noutput_items,
                                    gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = d_num_qam_symbols;
}

int discard_control_impl::general_work(int noutput_items,
                                       gr_vector_int& ninput_items,
                                       gr_vector_const_void_star& input_items,
                                       gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    memcpy(out, in+d_num_ctrl_symbols, d_num_data_symbols*sizeof(input_type));
        
    consume_each(d_num_qam_symbols);

    return d_num_data_symbols;
}

} /* namespace customModule */
} /* namespace gr */