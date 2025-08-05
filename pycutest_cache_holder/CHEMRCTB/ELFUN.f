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
C  Problem name : CHEMRCTB  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION T     , G     , BPLUS1, DADT  , D2ADT2
      DOUBLE PRECISION EX    , BEX   
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : ARR       
C
       T      = XVALUE(IELVAR(ILSTRT+     1))
       G      = EPVALU(IPSTRT+     1)
       BPLUS1 = EPVALU(IPSTRT+     2)
       DADT   = G / ( T * T )                            
       D2ADT2 = - 2.0 * DADT / T                         
       EX     = EXP( G - G / T )                         
       BEX    = EX * ( BPLUS1 - T )                      
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= BEX                                      
       ELSE
        FUVALS(IGSTRT+     1)= BEX * DADT - EX                          
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=BEX * ( DADT * DADT + D2ADT2 )           
     *                         - 2.0 * EX * DADT                        
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
