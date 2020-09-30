#ifndef __PYTHON_UTILS_H__
#define __PYTHON_UTILS_H__

#define PY_SSIZE_T_CLEAN // It is recommended to always define this before Python.h
#include <Python.h>
#include <vector>
#include <variant>
#include <unordered_map>
#include <typeinfo>
#include <iostream>

#include "utils.h"

namespace musher
{
    std::unordered_map<std::string, std::string> uMapCppToPythonTypes({
        {typeid(int).name(),         "i"},
        {typeid(uint32_t).name(),    "I"},
        {typeid(uint16_t).name(),    "H"},
        {typeid(uint8_t).name(),     "B"},
        {typeid(double).name(),      "d"},
        {typeid(float).name(),       "f"},
        {typeid(std::string).name(), "s"},
        {typeid(char*).name(),       "s"},
        {typeid(char).name(),        "C"},
    });

    /**
     * @brief convert basic c++ type to basic Pyobject type
     * @details Only allow arithmetic types into this function
     * 
     * @param var any arithmetic type
     * @return PyObject of the same type as c++ type
     */
    template< typename basicType,
        typename = std::enable_if_t<std::is_arithmetic<basicType>::value> >
    PyObject* basicTypeToPyobject(const basicType &var)
    {   
        /* Get type of variable passed in */
        std::string typeName = typeid(basicType).name();

        /* Bool has a special case */
        if(typeName == typeid(bool).name()){
            if(var)
                return Py_True;
            return Py_False;
        }

        /* Find the type of the variable in the unordered map */
        std::unordered_map<std::string, std::string>::const_iterator got = uMapCppToPythonTypes.find(typeName);

        if ( got == uMapCppToPythonTypes.end() )
        {
            std::string err_message = "Could not match C++ data type to Pyobject type";
            throw std::runtime_error(err_message);
        }

        return Py_BuildValue(got->second.c_str(), var);
    }

    /**
     * @brief convert basic c++ type to basic Pyobject type
     * @details strings have are a special case, so we overload the template function basicTypeToPyobject for the string case
     * 
     * @param var string type
     * @return PyObject of the same type as c++ type
     */
    PyObject* basicTypeToPyobject(const std::string &var)
    { 
        return Py_BuildValue("s", var.c_str());
    }

    /**
     * @brief Create a key, value pair that will be used to create a python dictionary.
     * 
     * @tparam T 
     * @param key Key name.
     * @param val Value name.
     * @return std::pair<PyObject*, PyObject*> A key, value tuple.
     */
    template <typename T>
    std::pair<PyObject*, PyObject*> createKVPair(const char* key, T val)
    {
        PyObject* pykey = basicTypeToPyobject(key);
        PyObject* pyval = basicTypeToPyobject(val);
        return std::make_pair(pykey, pyval);
    }

   /**
     * @brief Create a key, value pair where the val is already a PyObject
     * that will be used to create a python dictionary.
     * 
     * @tparam T 
     * @param key Key name.
     * @param val Value name.
     * @return std::pair<PyObject*, PyObject*> A key, value tuple.
     */
    std::pair<PyObject*, PyObject*> createKVPairFromPyObject(const char* key, PyObject* val)
    {
        PyObject* pykey = basicTypeToPyobject(key);
        return std::make_pair(pykey, val);
    }

    template<class T>
    T pyObjectToBasicType(PyObject* pyObj)
    {
        T basicType;
        /* Get type of template */
        std::string typeName = typeid(basicType).name();

        /* Find the type of the variable in the unordered map */
        std::unordered_map<std::string, std::string>::const_iterator got = uMapCppToPythonTypes.find(typeName);

        if ( got == uMapCppToPythonTypes.end() )
        {
            std::string err_message = "Could not match Pyobject type to C++ data type.";
            throw std::runtime_error(err_message);
        }

        std::string typeStr = got->second;

        /* Parse PyObject into corresponding c++ type */
        if(!PyArg_Parse(pyObj, typeStr.c_str(), &basicType))
        {
            PyTypeObject* type = pyObj->ob_type;
            const char* p = type->tp_name;

            std::string err_message = "Template of type ";
            err_message += "'";
            err_message += typeName;
            err_message += "'";
            err_message += " does not match PyObject of type ";
            err_message += "'";
            err_message += p;
            err_message += "'";
            err_message += ".";
            throw std::runtime_error(err_message);
        }

        return basicType;
    }

    /*  Accept a variant with any number of types
        The '...' will pack and unpack all variant types
    */
    template<typename ...types>
    PyObject* variantToPyobject(const std::variant<types...> &var)
    {   
        /* We define all posible types here because we want to be able to loop through a variant */
        PyObject* basicPyType;
        std::visit(
        overload(
            [&basicPyType](int arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](uint32_t arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](uint16_t arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](uint8_t arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](bool arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](double arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](float arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](std::string arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](char* arg) { basicPyType = basicTypeToPyobject(arg); },
            [&basicPyType](char arg) { basicPyType = basicTypeToPyobject(arg); },
            [](auto&&) { 
                std::string err_message = "Type in variant cannot be convert to PyObject";
                throw std::runtime_error(err_message); 
            }
        ),
        var
        );
        
        return basicPyType;
    }

    template<typename T>
    PyObject* vectorToList(const std::vector<T> &data) {
        PyObject* listObj = PyList_New( data.size() );

        for (unsigned int i = 0; i < data.size(); i++) {
            PyObject* item = basicTypeToPyobject(data[i]);
            if (!item) {
                Py_DECREF(listObj);
                throw std::logic_error("Unable to allocate memory for Python list");
            }
            PyList_SetItem(listObj, i, item);
        }
        return listObj;
    }


    template<typename T>
    std::vector<T> listToVector(PyObject* &listObj) {
        std::vector<T> data;
        if (!PyList_Check(listObj)) {
            std::string err_message = "PyObject passed is not a list";
            throw std::runtime_error(err_message); 
        }
        for(Py_ssize_t i = 0; i < PyList_Size(listObj); i++) {
            PyObject *value = PyList_GetItem(listObj, i);
            data.push_back( pyObjectToBasicType<T>(value) );
        }
        return data;
    }
}


#endif