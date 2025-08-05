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
C  Problem name : LSC1      
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , Y     , PX    , PY    , DX    
      DOUBLE PRECISION DY    , S     , SS    , S1    , S2    
      INTRINSIC SQRT  
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : S         
C
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       PX     = EPVALU(IPSTRT+     1)
       PY     = EPVALU(IPSTRT+     2)
       DX     = X - PX                                   
       DY     = Y - PY                                   
       SS     = DX * DX + DY * DY                        
       S      = SQRT(SS)                                 
       S1     = 1.0D0 / S                                
       S2     = - 1.0D0 / ( S * SS )                     
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= S                                        
       ELSE
        FUVALS(IGSTRT+     1)= S1 * DX                                  
        FUVALS(IGSTRT+     2)= S1 * DY                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=S2 * DX * DX + S1                        
         FUVALS(IHSTRT+     2)=S2 * DX * DY                             
         FUVALS(IHSTRT+     3)=S2 * DY * DY + S1                        
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
