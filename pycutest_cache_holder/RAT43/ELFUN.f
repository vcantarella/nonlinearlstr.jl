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
C  Problem name : RAT43     
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , V4    , X     
      DOUBLE PRECISION E     , V2MV3X, V4INV , V4INVP, E2    
      DOUBLE PRECISION EP1   , EP1L  , EP14  , EP14P1, EP14P2
      DOUBLE PRECISION VE    , VE2   , V42EPP, V42EP2, V42EP3
      INTRINSIC EXP   , LOG   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E         
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       V4     = XVALUE(IELVAR(ILSTRT+     4))
       X      = EPVALU(IPSTRT+     1)
       V2MV3X = V2 - V3 * X                              
       V4INV  = 1.0 / V4                                 
       V4INVP = V4INV + 1.0                              
       E      = EXP( V2MV3X )                            
       E2     = E * E                                    
       EP1    = E + 1.0                                  
       EP1L   = LOG( EP1 )                               
       EP14   = EP1 ** V4INV                             
       EP14P1 = EP1 ** V4INVP                            
       EP14P2 = EP1 ** ( V4INV + 2.0 )                   
       VE     = V4 * EP14P1                              
       VE2    = V4 * EP14P2                              
       V42EPP = EP14 * V4 ** 2                           
       V42EP2 = EP14P1 * V4 ** 2                         
       V42EP3 = EP14P1 * V4 ** 3                         
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 / EP14                                
       ELSE
        FUVALS(IGSTRT+     1)= 1.0 / EP14                               
        FUVALS(IGSTRT+     2)= - V1 * E / VE                            
        FUVALS(IGSTRT+     3)= V1 * X * E /VE                           
        FUVALS(IGSTRT+     4)= V1 * EP1L / V42EPP                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=- E /VE                                  
         FUVALS(IHSTRT+     4)=X * E / VE                               
         FUVALS(IHSTRT+     7)=EP1L / V42EPP                            
         FUVALS(IHSTRT+     3)=V1 * ( E2 * V4INVP / VE2 - E / VE )      
         FUVALS(IHSTRT+     5)=V1 * X * ( E / VE - E2 * V4INVP / VE2 )  
         FUVALS(IHSTRT+     8)=V1 * E * ( 1.0 / V42EP2 - EP1L / V42EP3 )
         FUVALS(IHSTRT+     6)=V1 * X ** 2                              
     *                          * ( E2 * V4INVP / VE2 - E / VE )        
         FUVALS(IHSTRT+     9)=V1 * X * E                               
     *                          * ( EP1L / V42EP3 - 1.0 / V42EP2 )      
         FUVALS(IHSTRT+    10)=( V1 / EP14) * ( EP1L ** 2 / V4 ** 4     
     *                            - 2.0 * EP1L / V4 ** 3  )             
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
