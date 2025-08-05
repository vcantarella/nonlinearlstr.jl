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
C  Problem name : LUKSAN22  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      INTEGER EXP   
      DOUBLE PRECISION X     , X1    , X2    , EXPARG
      IFSTAT = 0
      DO     4 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3
     *                                                        ), IELTYP
C
C  Element type : SQR       
C
    1  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * X                                    
       ELSE
        FUVALS(IGSTRT+     1)= X + X                                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0D0                                    
        END IF
       END IF
       GO TO     4
C
C  Element type : EXPDA     
C
    2  CONTINUE
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       X2     = XVALUE(IELVAR(ILSTRT+     2))
       X      =   X1    
     *          - X2    
       EXPARG = 2.0D0 * EXP( - X * X )                   
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= EXPARG                                   
       ELSE
        FUVALS(IGSTRT+     1)= - 2.0D0 * X * EXPARG                     
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=( 4.0D0 * X * X - 2.0D0 ) * EXPARG       
        END IF
       END IF
       GO TO     4
C
C  Element type : EXPDB     
C
    3  CONTINUE
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       X2     = XVALUE(IELVAR(ILSTRT+     2))
       X      =   X1    
     *          - X2    
       EXPARG = EXP( - 2.0D0 * X * X )                   
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= EXPARG                                   
       ELSE
        FUVALS(IGSTRT+     1)= - 4.0D0 * X * EXPARG                     
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=( 16.0D0 * X * X - 4.0D0 ) * EXPARG      
        END IF
       END IF
    4 CONTINUE
      RETURN
      END
