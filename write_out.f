      program write_out

      open(1,file='AERLOAD.OUT',STATUS='OLD')

      NTIME=1

      do i=1,NTIME
       read(1,*) R1,R2,R3,R4,R5,R6,R7,CL,R9,R10,R11,CM,CD,R14,R15,R16,
     .        R17,AA
      enddo

      close(1)

      open(1,file='clcd.out')
      write(1,*) AA,CL,CD,CM

      end
