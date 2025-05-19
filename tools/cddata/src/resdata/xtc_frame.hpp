#ifndef _XTC_FRAME_HPP
#define _XTC_FRAME_HPP

#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include <gromacs/pbcutil/pbc.h>

#include <xdrfile_xtc.h>

namespace resdata::xtc {

class Frame {
public:
  int natom;
  unsigned long nframe;
  int64_t *offsets;
  int step;
  float time;
  matrix box = {{0,0,0}, {0,0,0}, {0,0,0}};
  rvec *x;
  float prec;

  // create default constructor
  Frame() { x = (rvec*)malloc(0 * sizeof(rvec)); }
  Frame(int natom) : natom(natom) { x = (rvec*)malloc(natom * sizeof(rvec)); }

  int read_next_frame(XDRFILE *xd, bool nopbc, PbcType pbc_type, t_pbc *pbc, bool force_box)
  {
    int status = read_xtc(xd, natom, &step, &time, box, x, &prec);
    if (nopbc)
    {
      pbc = nullptr;
    }
    else
    {
      if (force_box) set_pbc(pbc, pbc_type, pbc->box);
      else set_pbc(pbc, pbc_type, box);
    }
    return status;    
  }
};

} // namespace resdata::xtc

#endif // _XTC_FRAME_HPP