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
C  Problem name : RAT42     
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , X     , E     
      DOUBLE PRECISION E2    , EP1   , EP12  , EP13  , V1E   
      DOUBLE PRECISION V1E2  
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E11       
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       X      = EPVALU(IPSTRT+     1)
       E      = EXP( V2 - V3 * X )                       
       E2     = E * E                                    
       EP1    = E + 1.0                                  
       EP12   = EP1 * EP1                                
       EP13   = EP1 * EP12                               
       V1E    = V1 * E                                   
       V1E2   = V1 * E2                                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 / EP1                                 
       ELSE
        FUVALS(IGSTRT+     1)= 1.0 / EP1                                
        FUVALS(IGSTRT+     2)= - V1E / EP12                             
        FUVALS(IGSTRT+     3)= V1E * X / EP12                           
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=- E / EP12                               
         FUVALS(IHSTRT+     4)=X * E / EP12                             
         FUVALS(IHSTRT+     3)=2.0 * V1E2 / EP13 - V1E / EP12           
         FUVALS(IHSTRT+     5)=( V1E / EP12 - 2.0 * V1E2 / EP13 ) * X   
         FUVALS(IHSTRT+     6)=( 2.0 * V1E2 / EP13 - V1E / EP12 )       
     *                           * X ** 2                               
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
