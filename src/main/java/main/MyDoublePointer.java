package main;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_double;

public class MyDoublePointer extends SWIGTYPE_p_double {
    public MyDoublePointer(long cPtr) {
        super(cPtr, false);
    }

    public long getCPtr() {
        return SWIGTYPE_p_double.getCPtr(this);
    }
}
