      SUBROUTINE RANGE( IELEMN, TRANSP, W1, W2, nelvar, ninvar,
     *                  itype, LW1, LW2 )
      INTEGER IELEMN, nelvar, ninvar, itype, LW1, LW2
      LOGICAL TRANSP
      DOUBLE PRECISION W1( LW1 ), W2( LW2 )
C
C  Problem name : LUKSAN16  
C
C  -- produced by SIFdecode 2.4
C
C  TRANSP = .FALSE. <=> W2 = U * W1
C  TRANSP = .TRUE.  <=> W2 = U(transpose) * W1
C
      INTEGER I
C
C  Element type : EXPSUM    
C
    1 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     1 ) *      2.00000
         W2(     3 ) =   W1(     1 ) *      3.00000
         W2(     4 ) =   W1(     1 ) *      4.00000
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 + W1(     2 ) *      2.00000
     *                 + W1(     3 ) *      3.00000
     *                 + W1(     4 ) *      4.00000
      END IF
      RETURN
      END
