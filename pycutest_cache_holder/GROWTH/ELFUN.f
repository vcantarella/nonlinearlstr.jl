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
C  Problem name : GROWTH    
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION U1    , U2    , U3    , RN    , LOGRN 
      DOUBLE PRECISION POWER 
      INTRINSIC LOG   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : FIT       
C
       U1     = XVALUE(IELVAR(ILSTRT+     1))
       U2     = XVALUE(IELVAR(ILSTRT+     2))
       U3     = XVALUE(IELVAR(ILSTRT+     3))
       RN     = EPVALU(IPSTRT+     1)
       LOGRN  = LOG( RN )                                
       POWER  = RN ** ( U2 + LOGRN * U3 )                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= U1 * POWER                               
       ELSE
        FUVALS(IGSTRT+     1)= POWER                                    
        FUVALS(IGSTRT+     2)= U1 * POWER * LOGRN                       
        FUVALS(IGSTRT+     3)= U1 * POWER * LOGRN ** 2                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=0.0                                      
         FUVALS(IHSTRT+     2)=POWER * LOGRN                            
         FUVALS(IHSTRT+     4)=POWER * LOGRN ** 2                       
         FUVALS(IHSTRT+     3)=U1 * POWER * LOGRN ** 2                  
         FUVALS(IHSTRT+     5)=U1 * POWER * LOGRN ** 3                  
         FUVALS(IHSTRT+     6)=U1 * POWER * LOGRN ** 4                  
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
