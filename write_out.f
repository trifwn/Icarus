      program write_out

         open(1,file='AERLOAD.OUT',STATUS='OLD')

         NTIME=1

         do i=1,NTIME
            read(1,*) R1,R2,R3,R4,R5,R6,R7,CL,CD,R10,R11,CM,R13,R14,
     .              R15,R16,R17,AA
c         TIME,DTHM,CLP,CDP,CNP,CTP,CMP,
c         CL ,CD ,CN ,CT ,CM , CM_FL,
c         CD_MOM, CN_MOM, CT_MOM,XSREAL,
c         AINFM+DTHM, XTRUPL, YTRUPL,
c         XTRLOL, YTRLOL, GONM,
c         EPSS(NTIME)
         enddo

         close(1)

         open(1,file='clcd.out')
         write(1,*) AA,CL,CD,CM

      end
