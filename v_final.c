#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define NR_ELEMENTE 200

void initialize_displs(int *displs, int *sendcounts, int group_nprocs)
{
    displs[0] = 0;
    for (int i = 1; i < group_nprocs; i++)
        displs[i] = displs[i - 1] + sendcounts[i - 1];
}

void initialize_sendcounts_vector(int *sendcounts, int group_nprocs) 
{
    int nr_elem_loc = NR_ELEMENTE / group_nprocs;
    int nr_elem_extra = NR_ELEMENTE % group_nprocs;

    for (int i = 0; i < group_nprocs; i++)
        sendcounts[i] = nr_elem_loc + (i < nr_elem_extra ? 1 : 0);
}

void initialize_sendcounts_matrix(int *sendcounts, int group_nprocs) 
{
    int nr_elem_loc = NR_ELEMENTE / group_nprocs;
    int nr_elem_extra = NR_ELEMENTE % group_nprocs;

    for (int i = 0; i < group_nprocs; i++)
        sendcounts[i] = nr_elem_loc * NR_ELEMENTE + (i < nr_elem_extra ? NR_ELEMENTE : 0);
}

double calculeaza_numarator(double *x, double *y, double *A, int n)
{
    double numarator = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < NR_ELEMENTE; j++)
        {
            numarator += x[i] * A[NR_ELEMENTE*i+j] * y[j];
        }
    }
    return numarator;
}

double calculeaza_numitor(double *x, double *y, int n)
{
    double numitor = 0;
    for(int i = 0; i < n; i++)
        numitor += x[i] * y[i];
    return numitor;
}

int citeste_vector(double **p_x, const char *nume_fisier, int nr_elemente)
{
    double *x;
    FILE *fp;
    
    fp = fopen(nume_fisier, "r");

    if(fp == NULL)
    {
        perror(nume_fisier);
        return -1;
    }
    x = (double*) malloc(nr_elemente*sizeof(double));
    if(x == NULL)
    {
        perror("malloc()");
        fclose(fp);
        return -1;
    }

    for(int i = 0; i < nr_elemente; i++)
    {
        if(fscanf(fp, "%lf", &x[i]) != 1)
        {
            fprintf(stderr, "problema cu fisierul %s\n", nume_fisier);
            fclose(fp);
            free(x);
            return -1;
        }
    }
    fclose(fp);
    *p_x = x;
    return 0;
}

int citeste_matrice(double **p_A, const char *nume_fisier, int nrows, int ncols)
{
    double *A;
    FILE *fp;

    fp = fopen(nume_fisier, "r");
    if(fp == NULL)
    {
        perror(nume_fisier);
        return -1;
    }
    A = (double*) malloc(nrows * ncols * sizeof(double));
    if(A == NULL)
    {
        perror("malloc()");
        return -1;
    }

    for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            if(fscanf(fp, "%lf", &A[nrows*i+j]) != 1)
            {
                fprintf(stderr, "problema cu fisierul %s\n", nume_fisier);
                free(A);
                fclose(fp);
                return -1;
            }
        }
    }
        
    fclose(fp);
    *p_A = A;
    return 0;
}

int main(int argc, char **argv)
{
    int nproc;
    int my_rank;
    int my_new_rank;
    int my_new_size;

    double *x = NULL;
    double *y = NULL;
    double *A = NULL;

    double *x_loc = NULL;
    double *y_loc = NULL;
    double *A_loc = NULL;

    double numarator = 0, numarator_loc = 0;
    double numitor = 0, numitor_loc = 0;

    MPI_Comm world_comm = MPI_COMM_WORLD, low_comm, high_comm;
    MPI_Group world_group, low_group, high_group;

    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(world_comm, &my_rank);
    MPI_Comm_size(world_comm, &nproc);

    if(nproc % 2 == 1)
    {
        if(my_rank == 0)
            fprintf(stderr, "nproc trebuie sa fie par\n");
        
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    int* low_ranks = (int*) malloc((nproc/2) * sizeof(int));
    int* high_ranks = (int*) malloc((nproc/2) * sizeof(int));
    for(int i = 0; i < nproc/2; i++)
    {
        low_ranks[i] = i;
        high_ranks[i] = i + nproc/2;
    }

    if(my_rank < nproc/2)
    {
        MPI_Comm_group(world_comm, &world_group);
        MPI_Group_incl(world_group, nproc/2, low_ranks, &low_group);
        MPI_Comm_create(world_comm, low_group, &low_comm);
        MPI_Group_rank(low_group, &my_new_rank);
        MPI_Group_size(low_group, &my_new_size);
    }

    if(my_rank >= nproc/2)
    {
        MPI_Comm_group(world_comm, &world_group);
        MPI_Group_incl(world_group, nproc/2, high_ranks, &high_group);
        MPI_Comm_create(world_comm, high_group, &high_comm);
        MPI_Group_rank(high_group, &my_new_rank);
        MPI_Group_size(high_group, &my_new_size);
    }

    if(my_new_rank == 0 && my_rank >= nproc/2)
    {
        if(citeste_vector(&x, "x.dat", NR_ELEMENTE) == -1)
        {
            exit(EXIT_FAILURE);
        }
        if(citeste_vector(&y, "y.dat", NR_ELEMENTE) == -1)
        {
            exit(EXIT_FAILURE);
        }
    }

    if(my_new_rank == 0 && my_rank < nproc/2)
    {
        if(citeste_vector(&x, "x.dat", NR_ELEMENTE) == -1)
        {
            exit(EXIT_FAILURE);
        }

        if(citeste_vector(&y, "y.dat", NR_ELEMENTE) == -1)
        {
            exit(EXIT_FAILURE);
        }

        if(citeste_matrice(&A, "mat.dat", NR_ELEMENTE, NR_ELEMENTE) == -1)
        {
            exit(EXIT_FAILURE);
        }
    }

    if(my_rank >= nproc/2)
    {
        int nr_elem_loc = NR_ELEMENTE / my_new_size;
        int nr_elem_extra = NR_ELEMENTE % my_new_size;

        int *sendcounts = (int*) malloc((nproc / 2) * sizeof(int));
        int *displs = (int*) malloc((nproc / 2) * sizeof(int));

        initialize_sendcounts_vector(sendcounts, nproc/2);
        initialize_displs(displs, sendcounts, nproc/2);

        x_loc = (double*) malloc(sendcounts[my_new_rank] * sizeof(double));
        y_loc = (double*) malloc(sendcounts[my_new_rank] * sizeof(double));

        MPI_Scatterv(x, sendcounts, displs, MPI_DOUBLE, x_loc, sendcounts[my_new_rank], MPI_DOUBLE, 0, high_comm);
        MPI_Scatterv(y, sendcounts, displs, MPI_DOUBLE, y_loc, sendcounts[my_new_rank], MPI_DOUBLE, 0, high_comm);

        numitor_loc = calculeaza_numitor(x_loc, y_loc, sendcounts[my_new_rank]);

        MPI_Reduce(&numitor_loc, &numitor, 1, MPI_DOUBLE, MPI_SUM, 0, high_comm);
    }

    if(my_rank < nproc/2)
    {
        int nr_elem_loc = NR_ELEMENTE / my_new_size;
        int nr_elem_extra = NR_ELEMENTE % my_new_size;

        int *sendcounts_x = (int*) malloc((nproc / 2) * sizeof(int));
        int *displs_x = (int*) malloc((nproc / 2) * sizeof(int));
        int *sendcounts_A = (int*) malloc((nproc / 2) * sizeof(int));
        int *displs_A = (int*) malloc((nproc / 2) * sizeof(int));

        initialize_sendcounts_vector(sendcounts_x, nproc/2);
        initialize_displs(displs_x, sendcounts_x, nproc/2);

        initialize_sendcounts_matrix(sendcounts_A, nproc/2);
        initialize_displs(displs_A, sendcounts_A, nproc/2);

        if(y == NULL)
        {
            y = (double*) malloc(NR_ELEMENTE * sizeof(double));
        }

        x_loc = (double*) malloc(sendcounts_x[my_new_rank] * sizeof(double));
        A_loc = (double*) malloc(sendcounts_A[my_new_rank] * sizeof(double));

        MPI_Scatterv(x, sendcounts_x, displs_x, MPI_DOUBLE, x_loc, sendcounts_x[my_new_rank], MPI_DOUBLE, 0, low_comm);
        MPI_Bcast(y, NR_ELEMENTE, MPI_DOUBLE, 0, low_comm);
        MPI_Scatterv(A, sendcounts_A, displs_A, MPI_DOUBLE, A_loc, sendcounts_A[my_new_rank], MPI_DOUBLE, 0, low_comm);
        
        numarator_loc = calculeaza_numarator(x_loc, y, A_loc, sendcounts_x[my_new_rank]);

        MPI_Reduce(&numarator_loc, &numarator, 1, MPI_DOUBLE, MPI_SUM, 0, low_comm);
    }

    if(my_new_rank == 0 && my_rank >= nproc/2)
    {
        MPI_Send(&numitor, 1, MPI_DOUBLE, 0, 1, world_comm);
    }

    if(my_rank == 0)
    {
        MPI_Recv(&numitor, 1, MPI_DOUBLE, nproc/2, 1, world_comm, MPI_STATUS_IGNORE);

        FILE *fp = fopen("result.txt", "w");
        fprintf(fp, "AVG = %.2lf", numarator/numitor);
        fclose(fp);
    }

    free(x);
    free(y);
    free(A);

    free(x_loc);
    free(y_loc);
    free(A_loc);

    MPI_Finalize();
    return 0;
}