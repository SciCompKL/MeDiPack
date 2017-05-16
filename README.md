# MeDiPack

[MeDiPack](http://www.scicomp.uni-kl.de/software/medi/) (Message Differentiation Package) is a tool that handles the MPI communication of Algorithmic Differentiation (AD) tools like
[CoDiPack](http://www.scicomp.uni-kl.de/software/codi/).

This is currently only a preview release of MeDiPack in order to make it available to partners and testers.
We hope that we can release the version 1.0 of MeDiPack during this summer.
Some of the features that we have in the 1.0 version of MeDiPack:
  - Full forward of the MPI 3.1 specification
  - AD handling for nearly all MPI functions
    - This includes:
      - Point to point communication
      - Collective communication
      - Blocking as well as non-blocking variants
    - This excludes (will be handled later):
      - One-Sided communication
      - IO functions

In the full release there will be a detailed specification of all functions, that are completely handled and which ones are just forwarded.

The [Scientific Computing Group](http://www.scicomp.uni-kl.de) at the TU Kaiserslautern develops MeDiPack and
will enhance and extend MeDiPack in the future. There is a newsletter available at [codi-info@uni-kl.de](https://lists.uni-kl.de/uni-kl/subscribe/codi-info)
(The newsletter for MeDiPack and CoDiPack is the same.)

