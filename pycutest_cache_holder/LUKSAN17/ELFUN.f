      SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, NCALCF, ITYPEE, 
     *                   ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, 
     *                   ICALCF, LTYPEE, LSTAEV, LELVAR, LNTVAR, 
     *                   LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, 
     *                   LEPVLU, IFFLAG, IFSTAT )
      INTEGER NCALCF, IFFLAG, LTYPEE, LSTAEV, LELVAR, LNTVAR
      INTEGER LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, LEPVLU
      INTEGER IFSTAT
      INTEGER ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
      INTEGER INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
      INTEGER ICALCF(LCALCF)
      DOUBLE PRECISION FUVALS(LFVALU), XVALUE(LXVALU), EPVALU(LEPVLU)
C
C  Problem name : LUKSAN17  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , A     , ASINX , ACOSX 
      INTRINSIC SIN   , COS   
      IFSTAT = 0
      DO     3 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2
     *                                                        ), IELTYP
C
C  Element type : ASINX     
C
    2  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       A      = EPVALU(IPSTRT+     1)
       ASINX  = A * SIN( X )                             
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= ASINX                                    
       ELSE
        FUVALS(IGSTRT+     1)= A * COS( X )                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- ASINX                                  
        END IF
       END IF
       GO TO     3
C
C  Element type : ACOSX     
C
    1  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       A      = EPVALU(IPSTRT+     1)
       ACOSX  = A * COS( X )                             
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= ACOSX                                    
       ELSE
        FUVALS(IGSTRT+     1)= - A * SIN( X )                           
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- ACOSX                                  
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
