#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=48, b0),"   /* For view and projection matrices */ \
"DescriptorTable(SRV(t0, numDescriptors=2) )"  /* SRV for particle positions, particle materials */