module params
    implicit none

    ! Time step length
    real(8), parameter :: dt = 0.001_8  ! DEFAULT
    !real(8), parameter :: dt = 0.00050_8

    ! State vector properties
    integer, parameter :: nx = 40
    integer, parameter :: ny = 10
    integer, parameter :: nz = 10
    integer, parameter :: n = nx+nx*ny+nx*ny*nz

    ! Model parameters
    real(8), parameter :: f = 20.0_8
    real(8), parameter :: h = 1.0_8
    real(8), parameter :: c = 10.0_8
    real(8), parameter :: b = 10.0_8
    real(8), parameter :: e = 10.0_8
    real(8), parameter :: d = 10.0_8
    real(8), parameter :: g_z = 1.0_8
end module params
