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
C  Problem name : LUKSAN16  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      INTEGER EXP   
      DOUBLE PRECISION S     , X1    , X2    , X3    , X4    
      DOUBLE PRECISION P2OL  , PLI   , EXPARG
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : EXPSUM    
C
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       X2     = XVALUE(IELVAR(ILSTRT+     2))
       X3     = XVALUE(IELVAR(ILSTRT+     3))
       X4     = XVALUE(IELVAR(ILSTRT+     4))
       P2OL   = EPVALU(IPSTRT+     1)
       PLI    = EPVALU(IPSTRT+     2)
       S      =   X1    
     *          + X2     *      2.00000
     *          + X3     *      3.00000
     *          + X4     *      4.00000
       EXPARG = P2OL * EXP( PLI * S )                    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= EXPARG                                   
       ELSE
        FUVALS(IGSTRT+     1)= PLI * EXPARG                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=PLI * PLI * EXPARG                       
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
